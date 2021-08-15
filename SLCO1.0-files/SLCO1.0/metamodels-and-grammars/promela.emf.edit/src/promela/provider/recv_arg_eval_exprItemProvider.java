/**
 */
package promela.provider;


import java.util.Collection;
import java.util.List;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ViewerNotification;

import promela.PromelaFactory;
import promela.PromelaPackage;
import promela.recv_arg_eval_expr;

/**
 * This is the item provider adapter for a {@link promela.recv_arg_eval_expr} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class recv_arg_eval_exprItemProvider
	extends recv_argItemProvider
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
	public recv_arg_eval_exprItemProvider(AdapterFactory adapterFactory) {
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

		}
		return itemPropertyDescriptors;
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
			childrenFeatures.add(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR);
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
	 * This returns recv_arg_eval_expr.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/recv_arg_eval_expr"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		return getString("_UI_recv_arg_eval_expr_type");
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

		switch (notification.getFeatureID(recv_arg_eval_expr.class)) {
			case PromelaPackage.RECV_ARG_EVAL_EXPR__EXPR:
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
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createvarref()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createbin_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createcond_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createlen_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createtimeout_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createnp__expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createenabled_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createpc_value_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createname_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createrun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createnum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createenum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RECV_ARG_EVAL_EXPR__EXPR,
				 PromelaFactory.eINSTANCE.createmtype_const()));
	}

}
