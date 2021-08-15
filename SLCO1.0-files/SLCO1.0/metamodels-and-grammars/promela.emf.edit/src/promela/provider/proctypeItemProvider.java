/**
 */
package promela.provider;


import java.util.Collection;
import java.util.List;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.ResourceLocator;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.ItemProviderAdapter;
import org.eclipse.emf.edit.provider.ViewerNotification;

import promela.PromelaFactory;
import promela.PromelaPackage;
import promela.proctype;

/**
 * This is the item provider adapter for a {@link promela.proctype} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class proctypeItemProvider
	extends ItemProviderAdapter
	implements
		IEditingDomainItemProvider,
		IStructuredItemContentProvider,
		ITreeItemContentProvider,
		IItemLabelProvider,
		IItemPropertySource {
	/**
	 * This constructs an instance from a factory and a notifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public proctypeItemProvider(AdapterFactory adapterFactory) {
		super(adapterFactory);
	}

	/**
	 * This returns the property descriptors for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public List<IItemPropertyDescriptor> getPropertyDescriptors(Object object) {
		if (itemPropertyDescriptors == null) {
			super.getPropertyDescriptors(object);

			addActivePropertyDescriptor(object);
			addNamePropertyDescriptor(object);
			addPriorityPropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Active feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addActivePropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_proctype_active_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_proctype_active_feature", "_UI_proctype_type"),
				 PromelaPackage.Literals.PROCTYPE__ACTIVE,
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.INTEGRAL_VALUE_IMAGE,
				 null,
				 null));
	}

	/**
	 * This adds a property descriptor for the Name feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addNamePropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_proctype_name_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_proctype_name_feature", "_UI_proctype_type"),
				 PromelaPackage.Literals.PROCTYPE__NAME,
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.GENERIC_VALUE_IMAGE,
				 null,
				 null));
	}

	/**
	 * This adds a property descriptor for the Priority feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addPriorityPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_proctype_priority_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_proctype_priority_feature", "_UI_proctype_type"),
				 PromelaPackage.Literals.PROCTYPE__PRIORITY,
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.INTEGRAL_VALUE_IMAGE,
				 null,
				 null));
	}

	/**
	 * This specifies how to implement {@link #getChildren} and is used to deduce an appropriate feature for an
	 * {@link org.eclipse.emf.edit.command.AddCommand}, {@link org.eclipse.emf.edit.command.RemoveCommand} or
	 * {@link org.eclipse.emf.edit.command.MoveCommand} in {@link #createCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Collection<? extends EStructuralFeature> getChildrenFeatures(Object object) {
		if (childrenFeatures == null) {
			super.getChildrenFeatures(object);
			childrenFeatures.add(PromelaPackage.Literals.PROCTYPE__ONE_DECLS);
			childrenFeatures.add(PromelaPackage.Literals.PROCTYPE__ENABLER);
			childrenFeatures.add(PromelaPackage.Literals.PROCTYPE__SEQUENCE);
		}
		return childrenFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EStructuralFeature getChildFeature(Object object, Object child) {
		// Check the type of the specified child object and return the proper feature to use for
		// adding (see {@link AddCommand}) it as a child.

		return super.getChildFeature(object, child);
	}

	/**
	 * This returns proctype.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/proctype"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		String label = ((proctype)object).getName();
		return label == null || label.length() == 0 ?
			getString("_UI_proctype_type") :
			getString("_UI_proctype_type") + " " + label;
	}

	/**
	 * This handles model notifications by calling {@link #updateChildren} to update any cached
	 * children and by creating a viewer notification, which it passes to {@link #fireNotifyChanged}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void notifyChanged(Notification notification) {
		updateChildren(notification);

		switch (notification.getFeatureID(proctype.class)) {
			case PromelaPackage.PROCTYPE__ACTIVE:
			case PromelaPackage.PROCTYPE__NAME:
			case PromelaPackage.PROCTYPE__PRIORITY:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), false, true));
				return;
			case PromelaPackage.PROCTYPE__ONE_DECLS:
			case PromelaPackage.PROCTYPE__ENABLER:
			case PromelaPackage.PROCTYPE__SEQUENCE:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), true, false));
				return;
		}
		super.notifyChanged(notification);
	}

	/**
	 * This adds {@link org.eclipse.emf.edit.command.CommandParameter}s describing the children
	 * that can be created under this object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void collectNewChildDescriptors(Collection<Object> newChildDescriptors, Object object) {
		super.collectNewChildDescriptors(newChildDescriptors, object);

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ONE_DECLS,
				 PromelaFactory.eINSTANCE.createone_decl()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createvarref()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createbin_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createcond_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createlen_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createtimeout_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createnp__expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createenabled_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createpc_value_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createname_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createrun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createandor_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createchanpoll_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createnum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createenum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__ENABLER,
				 PromelaFactory.eINSTANCE.createmtype_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createone_decl()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createchanassert()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createvarref()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createsend()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createreceive()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createassign_std()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createassign_inc()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createassign_dec()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createif_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createdo_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createatomic_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.created_step_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createblock_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createelse_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createbreak_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.creategoto_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createprint_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createassert_stmnt()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createc_code()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createc_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createc_decl()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createc_track()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createc_state()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createbin_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createcond_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createlen_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createtimeout_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createnp__expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createenabled_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createpc_value_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createname_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createrun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createandor_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createchanpoll_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createnum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createenum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.PROCTYPE__SEQUENCE,
				 PromelaFactory.eINSTANCE.createmtype_const()));
	}

	/**
	 * This returns the label text for {@link org.eclipse.emf.edit.command.CreateChildCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getCreateChildText(Object owner, Object feature, Object child, Collection<?> selection) {
		Object childFeature = feature;
		Object childObject = child;

		boolean qualify =
			childFeature == PromelaPackage.Literals.PROCTYPE__ONE_DECLS ||
			childFeature == PromelaPackage.Literals.PROCTYPE__SEQUENCE ||
			childFeature == PromelaPackage.Literals.PROCTYPE__ENABLER;

		if (qualify) {
			return getString
				("_UI_CreateChild_text2",
				 new Object[] { getTypeText(childObject), getFeatureText(childFeature), getTypeText(owner) });
		}
		return super.getCreateChildText(owner, feature, child, selection);
	}

	/**
	 * Return the resource locator for this item provider's resources.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public ResourceLocator getResourceLocator() {
		return PromelaEditPlugin.INSTANCE;
	}

}
