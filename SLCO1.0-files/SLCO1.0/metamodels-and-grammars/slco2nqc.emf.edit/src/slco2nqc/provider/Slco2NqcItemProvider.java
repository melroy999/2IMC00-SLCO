/**
 */
package slco2nqc.provider;


import java.util.Collection;
import java.util.List;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.ResourceLocator;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemProviderAdapter;
import org.eclipse.emf.edit.provider.ViewerNotification;

import slco2nqc.Slco2Nqc;
import slco2nqc.Slco2nqcFactory;
import slco2nqc.Slco2nqcPackage;

/**
 * This is the item provider adapter for a {@link slco2nqc.Slco2Nqc} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class Slco2NqcItemProvider
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
	public Slco2NqcItemProvider(AdapterFactory adapterFactory) {
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
			childrenFeatures.add(Slco2nqcPackage.Literals.SLCO2_NQC__SIGNAL2_MOTOR_LIST);
			childrenFeatures.add(Slco2nqcPackage.Literals.SLCO2_NQC__SIGNAL2_SENSOR_LIST);
			childrenFeatures.add(Slco2nqcPackage.Literals.SLCO2_NQC__MOTOR_INIT_LIST);
			childrenFeatures.add(Slco2nqcPackage.Literals.SLCO2_NQC__CLASS_LIST);
			childrenFeatures.add(Slco2nqcPackage.Literals.SLCO2_NQC__SENSOR_INIT_LIST);
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
	 * This returns Slco2Nqc.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/Slco2Nqc"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		return getString("_UI_Slco2Nqc_type");
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

		switch (notification.getFeatureID(Slco2Nqc.class)) {
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST:
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST:
			case Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST:
			case Slco2nqcPackage.SLCO2_NQC__CLASS_LIST:
			case Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST:
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
				(Slco2nqcPackage.Literals.SLCO2_NQC__SIGNAL2_MOTOR_LIST,
				 Slco2nqcFactory.eINSTANCE.createSignal2MotorList()));

		newChildDescriptors.add
			(createChildParameter
				(Slco2nqcPackage.Literals.SLCO2_NQC__SIGNAL2_SENSOR_LIST,
				 Slco2nqcFactory.eINSTANCE.createSignal2SensorList()));

		newChildDescriptors.add
			(createChildParameter
				(Slco2nqcPackage.Literals.SLCO2_NQC__MOTOR_INIT_LIST,
				 Slco2nqcFactory.eINSTANCE.createMotorInitList()));

		newChildDescriptors.add
			(createChildParameter
				(Slco2nqcPackage.Literals.SLCO2_NQC__CLASS_LIST,
				 Slco2nqcFactory.eINSTANCE.createClassList()));

		newChildDescriptors.add
			(createChildParameter
				(Slco2nqcPackage.Literals.SLCO2_NQC__SENSOR_INIT_LIST,
				 Slco2nqcFactory.eINSTANCE.createSensorInitList()));
	}

	/**
	 * Return the resource locator for this item provider's resources.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public ResourceLocator getResourceLocator() {
		return Slco2nqcEditPlugin.INSTANCE;
	}

}
